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


# =============================================================================
# EMERGENCE DETECTION (AGI Upgrade)
# =============================================================================

@dataclass
class EmergenceTrigger:
    """
    A trigger condition for detecting emergent behaviors.

    Emergence = collective behavior that isn't explicitly programmed,
    arising from interactions between agents/systems.
    """
    trigger_id: str
    name: str
    description: str

    # Detection parameters
    pattern_type: str  # "threshold", "anomaly", "correlation", "convergence"
    metric_name: str  # What to measure
    threshold_value: float = 0.7
    window_seconds: float = 300.0  # Time window for detection

    # Response configuration
    response_action: str  # "notify", "amplify", "dampen", "record"
    response_payload: dict = field(default_factory=dict)

    # State tracking
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    cooldown_seconds: float = 60.0  # Min time between triggers

    def can_trigger(self) -> bool:
        """Check if trigger is off cooldown."""
        if not self.is_active:
            return False
        if not self.last_triggered:
            return True
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds


@dataclass
class EmergenceEvent:
    """Record of a detected emergence event."""
    event_id: str
    trigger_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # What was detected
    pattern_description: str = ""
    metric_values: dict = field(default_factory=dict)
    contributing_agents: list[str] = field(default_factory=list)

    # Classification
    emergence_type: str = "unknown"  # "synergy", "consensus", "innovation", "cascade"
    novelty_score: float = 0.0  # How unexpected was this
    impact_score: float = 0.0  # How significant

    # Response taken
    response_action: str = ""
    response_result: Optional[str] = None


@dataclass
class EmergenceMetrics:
    """Metrics tracked for emergence detection."""
    # Synergy metrics
    collective_vs_individual_ratio: float = 1.0  # >1 = synergy
    cross_agent_correlation: float = 0.0  # How aligned are agents

    # Consensus metrics
    opinion_convergence: float = 0.0  # Are agents converging on answers
    decision_agreement_rate: float = 0.0

    # Innovation metrics
    novel_pattern_count: int = 0
    solution_diversity: float = 0.0  # Variety of approaches

    # Cascade metrics
    influence_spread_rate: float = 0.0  # How fast ideas spread
    activation_chains: int = 0  # Sequences of agent activations

    # Timestamp
    measured_at: datetime = field(default_factory=datetime.now)


class ProactiveState(Enum):
    IDLE = "idle"
    OBSERVING = "observing"  # Watching user actions to learn
    ANALYZING = "analyzing"  # Analyzing context to find opportunities
    SUGGESTING = "suggesting" # Preparing a suggestion
    ACTING = "acting"        # Autonomously executing a task
    DETECTING_EMERGENCE = "detecting_emergence"  # AGI: Monitoring for emergence


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
            elif task.schedule_type == "one_off":
                if task.next_run and now >= task.next_run and not task.last_run:
                    should_run = True

            if should_run:
                logger.info(f"Running scheduled task: {task.description}")
                self.state = ProactiveState.ACTING
                try:
                    # Execute the task via planner
                    result = await self._execute_scheduled_task(task)
                    task.last_run = now

                    # Disable one-off tasks after execution
                    if task.schedule_type == "one_off":
                        task.enabled = False

                    logger.info(f"Scheduled task completed: {task.id} - Success: {result.get('success', False)}")

                except Exception as e:
                    logger.error(f"Failed to run scheduled task {task.id}: {e}")
                    # Still mark as run to prevent infinite retries
                    task.last_run = now

                self.state = ProactiveState.IDLE

    async def _execute_scheduled_task(self, task: ScheduledTask) -> dict:
        """
        Execute a scheduled task using the planner or LLM.

        Args:
            task: The scheduled task to execute

        Returns:
            Result dictionary with success status and output
        """
        try:
            # If we have an LLM function, use it to determine the best action
            if self.llm_fn:
                prompt = f"""Execute the following scheduled task for the Farnsworth proactive agent system.

TASK DETAILS:
- Description: {task.description}
- Task ID: {task.id}
- Schedule Type: {task.schedule_type}
- Current Time: {datetime.now().isoformat()}

EXECUTION GUIDELINES:
1. Analyze what the task requires
2. Determine if it's an information gathering, state change, or notification task
3. Execute the appropriate action
4. Report results accurately

CONSTRAINTS:
- Do NOT modify critical system files without explicit permission in task description
- Do NOT make external API calls unless task specifically requires it
- Do NOT access user personal data beyond what's in the task context
- Maximum execution time assumption: 30 seconds

OUTPUT FORMAT (JSON only):
{{
    "success": true | false,
    "action_taken": "specific description of what was done",
    "output": "relevant result data or message",
    "side_effects": ["list of any state changes made"] | null,
    "error": "error message if success is false" | null
}}

Execute the task and return ONLY the JSON response:"""
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                # Parse JSON response
                try:
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start >= 0 and end > start:
                        result = json.loads(response[start:end])
                        return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Task response not JSON-formatted, treating as plain text: {e}")

                return {"success": True, "action_taken": "processed", "output": response}

            # Fallback: Use planner if available
            if self.planner and hasattr(self.planner, 'create_plan'):
                plan_id = await self.planner.create_plan(
                    goal=task.description,
                    context={"scheduled_task_id": task.id}
                )
                return {"success": True, "action_taken": "plan_created", "plan_id": plan_id}

            # No execution method available
            logger.warning(f"No execution method available for task {task.id}")
            return {"success": False, "error": "No execution method available"}

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {"success": False, "error": str(e)}

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
            
        # 5. Novel Context Attributes - Mood and Focus Analysis
        mood, focus_score = await self._analyze_sentiment(recent_context)
            
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

        prompt = f"""Analyze the user's context and suggest ONE proactive action if genuinely helpful.

CURRENT STATE:
- Time: {context['timestamp']}
- Work Hours: {context['is_work_hours']}
- Activity Level: {context['activity_level']:.2f}
- Active Plan: {json.dumps(context.get('active_plan', {}), default=str)}

RECENT CONVERSATION:
{context['recent_context'][:2000]}

ANALYSIS CRITERIA:
1. Is there a CLEAR, SPECIFIC need the user hasn't addressed?
2. Would the suggestion save significant time (>5 min of user effort)?
3. Is the suggestion actionable immediately?
4. Does it align with the user's current focus/project?

SUGGESTION TYPES:
- "run_task": Execute a specific automated task (file creation, data processing)
- "plan_task": Create a multi-step plan for a complex goal
- "search": Find relevant information or code examples
- "none": No action needed (return empty object)

WHEN TO RETURN EMPTY {{}}:
- User is in flow state (high activity, focused conversation)
- No clear unmet need
- Last suggestion was < 10 minutes ago
- Context is ambiguous

OUTPUT FORMAT (JSON only):
{{
    "title": "5-10 word action title",
    "description": "1-2 sentences explaining the benefit",
    "action_type": "run_task" | "plan_task" | "search",
    "action_details": "specific parameters for the action",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this helps now"
}}

Only suggest if confidence >= 0.7. Return {{}} otherwise."""
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

    async def _analyze_sentiment(self, context: str) -> tuple[str, float]:
        """
        Analyze sentiment and focus level from context.

        Args:
            context: Recent conversation context

        Returns:
            Tuple of (mood, focus_score)
        """
        context_lower = context.lower()

        # Use LLM for sentiment analysis if available
        if self.llm_fn and len(context) > 50:
            try:
                prompt = f"""Analyze the emotional state and focus level of this conversation.

CONVERSATION CONTEXT:
{context[:1000]}

MOOD DEFINITIONS:
- "frustrated": User encountering repeated errors, expressing annoyance, using negative language
- "focused": Deep in task execution, short precise messages, technical vocabulary
- "exploratory": Asking questions, trying alternatives, open-ended queries
- "confused": Asking for clarification, misunderstanding responses, uncertain language
- "satisfied": Expressing gratitude, acknowledging completion, positive language
- "neutral": Normal working state, no strong emotional indicators

FOCUS SCORE CRITERIA:
- 0.0-0.3: Casual/distracted (small talk, tangential topics)
- 0.4-0.6: Normal working (steady progress, some context switching)
- 0.7-0.9: High focus (deep work, rapid iteration, detailed questions)
- 1.0: Peak intensity (urgent deadline, critical bug, time pressure)

SIGNALS TO LOOK FOR:
- Message frequency and length
- Technical depth of questions
- Emotional language (exclamations, capitalization)
- Request urgency indicators
- Context switching patterns

OUTPUT FORMAT (JSON only):
{{
    "mood": "frustrated" | "focused" | "exploratory" | "confused" | "satisfied" | "neutral",
    "focus_score": 0.0-1.0,
    "primary_signals": ["signal1", "signal2"],
    "reasoning": "1 sentence explanation"
}}

Analyze and return ONLY the JSON:"""

                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                # Parse JSON response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                    return data.get("mood", "neutral"), data.get("focus_score", 0.5)

            except Exception as e:
                logger.debug(f"LLM sentiment analysis failed, using heuristics: {e}")

        # Fallback to heuristic analysis
        focus_score = 0.5
        mood = "neutral"

        # Frustration indicators
        frustration_words = ["error", "fail", "broken", "bug", "issue", "problem", "wrong", "crash", "exception"]
        if any(word in context_lower for word in frustration_words):
            mood = "frustrated"
            focus_score = 0.9

        # Confusion indicators
        confusion_words = ["confused", "don't understand", "what does", "how do", "why is", "unclear"]
        if any(word in context_lower for word in confusion_words):
            mood = "confused"
            focus_score = 0.7

        # Satisfaction indicators
        satisfaction_words = ["thanks", "perfect", "great", "awesome", "works", "solved", "fixed"]
        if any(word in context_lower for word in satisfaction_words):
            mood = "satisfied"
            focus_score = 0.3

        # Exploration indicators (long queries, many questions)
        if context_lower.count("?") > 3 or len(context) > 500:
            mood = "exploratory"
            focus_score = 0.6

        return mood, focus_score

    def add_scheduled_task(
        self,
        description: str,
        schedule_type: str = "interval",
        schedule_value: Any = 3600,
        task_id: Optional[str] = None,
    ) -> ScheduledTask:
        """
        Add a new scheduled task.

        Args:
            description: What the task should do
            schedule_type: "interval", "cron", or "one_off"
            schedule_value: Seconds for interval, cron string, or datetime for one_off
            task_id: Optional custom ID

        Returns:
            The created ScheduledTask
        """
        task = ScheduledTask(
            id=task_id or f"task_{len(self.scheduled_tasks) + 1}_{int(datetime.now().timestamp())}",
            description=description,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            next_run=schedule_value if schedule_type == "one_off" and isinstance(schedule_value, datetime) else None,
        )
        self.scheduled_tasks.append(task)
        logger.info(f"Added scheduled task: {task.id} - {description}")
        return task

    def remove_scheduled_task(self, task_id: str) -> bool:
        """Remove a scheduled task by ID."""
        for i, task in enumerate(self.scheduled_tasks):
            if task.id == task_id:
                self.scheduled_tasks.pop(i)
                logger.info(f"Removed scheduled task: {task_id}")
                return True
        return False

    def get_pending_suggestions(self) -> list[Suggestion]:
        """Get all non-dismissed suggestions."""
        return [s for s in self.suggestions if not s.is_dismissed]

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion."""
        for s in self.suggestions:
            if s.id == suggestion_id:
                s.is_dismissed = True
                return True
        return False

    def accept_suggestion(self, suggestion_id: str) -> Optional[Suggestion]:
        """Accept a suggestion and return it for execution."""
        for s in self.suggestions:
            if s.id == suggestion_id:
                s.is_accepted = True
                return s
        return None

    def get_status(self) -> dict:
        """Get proactive agent status."""
        return {
            "state": self.state.value,
            "is_running": self._is_running,
            "suggestion_count": len(self.suggestions),
            "pending_suggestions": len(self.get_pending_suggestions()),
            "scheduled_tasks": len(self.scheduled_tasks),
            "active_scheduled_tasks": len([t for t in self.scheduled_tasks if t.enabled]),
        }

    # =========================================================================
    # EMERGENCE DETECTION (AGI Upgrade)
    # =========================================================================

    def _init_emergence_tracking(self):
        """Initialize emergence detection state."""
        if not hasattr(self, '_emergence_triggers'):
            self._emergence_triggers: list[EmergenceTrigger] = []
            self._emergence_events: list[EmergenceEvent] = []
            self._metrics_history: list[EmergenceMetrics] = []
            self._agent_states: dict[str, list[dict]] = {}  # agent_id -> state history
            self._emergence_callbacks: list[Callable] = []

            # Register default triggers
            self._register_default_triggers()

    def _register_default_triggers(self):
        """Register default emergence detection triggers."""
        self._emergence_triggers = [
            EmergenceTrigger(
                trigger_id="synergy_detection",
                name="Synergy Emergence",
                description="Collective output exceeds sum of individual contributions",
                pattern_type="threshold",
                metric_name="collective_vs_individual_ratio",
                threshold_value=1.2,  # 20% better than individuals
                response_action="amplify",
                response_payload={"boost_collaboration": True},
            ),
            EmergenceTrigger(
                trigger_id="consensus_formation",
                name="Consensus Formation",
                description="Agents converging on similar conclusions independently",
                pattern_type="convergence",
                metric_name="opinion_convergence",
                threshold_value=0.8,
                response_action="record",
                response_payload={"log_consensus": True},
            ),
            EmergenceTrigger(
                trigger_id="innovation_spike",
                name="Innovation Spike",
                description="Unusual increase in novel solution patterns",
                pattern_type="anomaly",
                metric_name="novel_pattern_count",
                threshold_value=3.0,  # 3x baseline
                response_action="notify",
                response_payload={"alert_type": "innovation"},
            ),
            EmergenceTrigger(
                trigger_id="cascade_activation",
                name="Cascade Activation",
                description="Chain reaction of agent activations",
                pattern_type="correlation",
                metric_name="activation_chains",
                threshold_value=5,  # 5+ chain length
                response_action="dampen",
                response_payload={"rate_limit": True},
            ),
        ]

    def add_emergence_trigger(self, trigger: EmergenceTrigger):
        """Add a custom emergence trigger."""
        self._init_emergence_tracking()
        self._emergence_triggers.append(trigger)
        logger.info(f"Added emergence trigger: {trigger.name}")

    def on_emergence(self, callback: Callable[[EmergenceEvent], None]):
        """Register a callback for emergence events."""
        self._init_emergence_tracking()
        self._emergence_callbacks.append(callback)

    async def record_agent_state(
        self,
        agent_id: str,
        state: dict,
    ):
        """
        Record an agent's state for emergence analysis.

        Args:
            agent_id: The agent identifier
            state: Current state dict (should include 'output', 'confidence', 'timestamp')
        """
        self._init_emergence_tracking()

        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = []

        state['recorded_at'] = datetime.now()
        self._agent_states[agent_id].append(state)

        # Keep only recent states (last 100 per agent)
        if len(self._agent_states[agent_id]) > 100:
            self._agent_states[agent_id] = self._agent_states[agent_id][-100:]

    async def compute_emergence_metrics(
        self,
        time_window_seconds: float = 300.0,
    ) -> EmergenceMetrics:
        """
        Compute current emergence metrics from agent states.

        Args:
            time_window_seconds: Look at states within this window

        Returns:
            EmergenceMetrics snapshot
        """
        self._init_emergence_tracking()

        metrics = EmergenceMetrics()
        cutoff = datetime.now()

        # Gather recent states
        recent_states: dict[str, list[dict]] = {}
        for agent_id, states in self._agent_states.items():
            recent = [
                s for s in states
                if (cutoff - s.get('recorded_at', cutoff)).total_seconds() < time_window_seconds
            ]
            if recent:
                recent_states[agent_id] = recent

        if len(recent_states) < 2:
            # Not enough data for emergence analysis
            return metrics

        # 1. Synergy: Compare collective outputs to individual
        individual_scores = []
        collective_indicators = []
        for agent_id, states in recent_states.items():
            for state in states:
                if 'confidence' in state:
                    individual_scores.append(state['confidence'])
                if 'collective_contribution' in state:
                    collective_indicators.append(state['collective_contribution'])

        if individual_scores:
            avg_individual = sum(individual_scores) / len(individual_scores)
            if collective_indicators:
                avg_collective = sum(collective_indicators) / len(collective_indicators)
                metrics.collective_vs_individual_ratio = avg_collective / avg_individual if avg_individual > 0 else 1.0

        # 2. Consensus: Check for convergence in outputs
        outputs_by_topic: dict[str, list[str]] = {}
        for agent_id, states in recent_states.items():
            for state in states:
                topic = state.get('topic', 'general')
                output = state.get('output', '')
                if topic not in outputs_by_topic:
                    outputs_by_topic[topic] = []
                outputs_by_topic[topic].append(output)

        if outputs_by_topic:
            convergence_scores = []
            for topic, outputs in outputs_by_topic.items():
                if len(outputs) >= 2:
                    # Simple similarity check (word overlap)
                    all_words = set()
                    for o in outputs:
                        all_words.update(o.lower().split())
                    if all_words:
                        overlap_ratio = self._compute_output_similarity(outputs)
                        convergence_scores.append(overlap_ratio)

            if convergence_scores:
                metrics.opinion_convergence = sum(convergence_scores) / len(convergence_scores)

        # 3. Innovation: Count novel patterns
        seen_patterns: set[str] = set()
        novel_count = 0
        for agent_id, states in recent_states.items():
            for state in states:
                pattern = state.get('solution_pattern', '')
                if pattern and pattern not in seen_patterns:
                    seen_patterns.add(pattern)
                    novel_count += 1

        metrics.novel_pattern_count = novel_count
        metrics.solution_diversity = len(seen_patterns) / max(1, sum(len(s) for s in recent_states.values()))

        # 4. Cascade: Detect activation chains
        activation_times: list[tuple[str, datetime]] = []
        for agent_id, states in recent_states.items():
            for state in states:
                if state.get('activated', False):
                    activation_times.append((agent_id, state.get('recorded_at', cutoff)))

        activation_times.sort(key=lambda x: x[1])
        chain_length = self._detect_activation_chain(activation_times)
        metrics.activation_chains = chain_length

        # Compute influence spread rate
        if len(activation_times) >= 2:
            time_span = (activation_times[-1][1] - activation_times[0][1]).total_seconds()
            if time_span > 0:
                metrics.influence_spread_rate = len(activation_times) / time_span

        self._metrics_history.append(metrics)

        # Keep only recent metrics (last 50)
        if len(self._metrics_history) > 50:
            self._metrics_history = self._metrics_history[-50:]

        return metrics

    def _compute_output_similarity(self, outputs: list[str]) -> float:
        """Compute average pairwise similarity between outputs."""
        if len(outputs) < 2:
            return 0.0

        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                words_i = set(outputs[i].lower().split())
                words_j = set(outputs[j].lower().split())
                if words_i or words_j:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    similarities.append(intersection / union if union > 0 else 0)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _detect_activation_chain(
        self,
        activations: list[tuple[str, datetime]],
        max_gap_seconds: float = 5.0,
    ) -> int:
        """Detect longest chain of rapid activations."""
        if not activations:
            return 0

        max_chain = 1
        current_chain = 1

        for i in range(1, len(activations)):
            gap = (activations[i][1] - activations[i - 1][1]).total_seconds()
            if gap <= max_gap_seconds:
                current_chain += 1
                max_chain = max(max_chain, current_chain)
            else:
                current_chain = 1

        return max_chain

    async def check_emergence_triggers(
        self,
        metrics: Optional[EmergenceMetrics] = None,
    ) -> list[EmergenceEvent]:
        """
        Check all triggers against current metrics.

        Args:
            metrics: Pre-computed metrics, or compute fresh if None

        Returns:
            List of triggered emergence events
        """
        self._init_emergence_tracking()

        if metrics is None:
            metrics = await self.compute_emergence_metrics()

        triggered_events = []

        for trigger in self._emergence_triggers:
            if not trigger.can_trigger():
                continue

            triggered = False
            metric_value = getattr(metrics, trigger.metric_name, None)

            if metric_value is None:
                continue

            # Check based on pattern type
            if trigger.pattern_type == "threshold":
                triggered = metric_value >= trigger.threshold_value

            elif trigger.pattern_type == "anomaly":
                # Compare to historical baseline
                baseline = self._get_metric_baseline(trigger.metric_name)
                if baseline > 0:
                    triggered = metric_value >= baseline * trigger.threshold_value

            elif trigger.pattern_type == "convergence":
                triggered = metric_value >= trigger.threshold_value

            elif trigger.pattern_type == "correlation":
                triggered = metric_value >= trigger.threshold_value

            if triggered:
                event = await self._create_emergence_event(trigger, metrics, metric_value)
                triggered_events.append(event)

                # Execute response
                await self._execute_emergence_response(trigger, event)

                # Update trigger state
                trigger.last_triggered = datetime.now()
                trigger.trigger_count += 1

                logger.info(
                    f"Emergence detected: {trigger.name} "
                    f"(value={metric_value:.3f}, threshold={trigger.threshold_value:.3f})"
                )

        return triggered_events

    def _get_metric_baseline(self, metric_name: str) -> float:
        """Get historical baseline for a metric."""
        if not self._metrics_history:
            return 0.0

        values = [
            getattr(m, metric_name, 0)
            for m in self._metrics_history[:-1]  # Exclude most recent
        ]

        return sum(values) / len(values) if values else 0.0

    async def _create_emergence_event(
        self,
        trigger: EmergenceTrigger,
        metrics: EmergenceMetrics,
        triggered_value: float,
    ) -> EmergenceEvent:
        """Create an emergence event record."""
        event = EmergenceEvent(
            event_id=f"emergence_{len(self._emergence_events) + 1}_{int(datetime.now().timestamp())}",
            trigger_id=trigger.trigger_id,
            pattern_description=trigger.description,
            metric_values={
                trigger.metric_name: triggered_value,
                "collective_ratio": metrics.collective_vs_individual_ratio,
                "convergence": metrics.opinion_convergence,
                "novel_patterns": metrics.novel_pattern_count,
            },
            contributing_agents=list(self._agent_states.keys()),
            emergence_type=self._classify_emergence(trigger, metrics),
            novelty_score=self._compute_novelty_score(trigger, triggered_value),
            impact_score=self._compute_impact_score(trigger, metrics),
        )

        self._emergence_events.append(event)

        # Notify callbacks
        for callback in self._emergence_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Emergence callback error: {e}")

        return event

    def _classify_emergence(self, trigger: EmergenceTrigger, metrics: EmergenceMetrics) -> str:
        """Classify the type of emergence."""
        if trigger.metric_name == "collective_vs_individual_ratio":
            return "synergy"
        elif trigger.metric_name == "opinion_convergence":
            return "consensus"
        elif trigger.metric_name == "novel_pattern_count":
            return "innovation"
        elif trigger.metric_name == "activation_chains":
            return "cascade"
        return "unknown"

    def _compute_novelty_score(self, trigger: EmergenceTrigger, value: float) -> float:
        """Compute how novel this emergence event is."""
        baseline = self._get_metric_baseline(trigger.metric_name)
        if baseline == 0:
            return 0.5  # Unknown baseline

        ratio = value / baseline
        # Novelty increases logarithmically with ratio
        import math
        return min(1.0, math.log(max(1, ratio)) / 2)

    def _compute_impact_score(self, trigger: EmergenceTrigger, metrics: EmergenceMetrics) -> float:
        """Compute the potential impact of this emergence."""
        # Impact based on number of agents involved and metric magnitude
        agent_factor = min(1.0, len(self._agent_states) / 10)

        # Combine multiple metrics for impact
        metric_factor = (
            metrics.collective_vs_individual_ratio * 0.3 +
            metrics.opinion_convergence * 0.2 +
            metrics.novel_pattern_count / 10 * 0.3 +
            metrics.influence_spread_rate * 0.2
        )

        return min(1.0, agent_factor * 0.4 + metric_factor * 0.6)

    async def _execute_emergence_response(
        self,
        trigger: EmergenceTrigger,
        event: EmergenceEvent,
    ):
        """Execute the response action for a triggered emergence."""
        action = trigger.response_action
        payload = trigger.response_payload

        event.response_action = action

        try:
            if action == "notify":
                # Create a high-priority suggestion
                await self._generate_suggestion({
                    "title": f"Emergence Detected: {trigger.name}",
                    "description": f"{trigger.description}. Novelty: {event.novelty_score:.2f}, Impact: {event.impact_score:.2f}",
                    "action_type": "none",
                    "confidence": event.impact_score,
                    "reasoning": "Automatic emergence detection",
                })
                event.response_result = "notification_created"

            elif action == "amplify":
                # Could adjust system parameters to encourage more of this behavior
                logger.info(f"Amplifying emergence: {trigger.name}")
                event.response_result = "amplification_logged"

            elif action == "dampen":
                # Could add rate limiting or cooling periods
                logger.info(f"Dampening emergence: {trigger.name}")
                event.response_result = "dampening_logged"

            elif action == "record":
                # Just log for analysis
                event.response_result = "recorded"

        except Exception as e:
            logger.error(f"Emergence response error: {e}")
            event.response_result = f"error: {e}"

    def get_emergence_history(
        self,
        limit: int = 20,
        emergence_type: Optional[str] = None,
    ) -> list[EmergenceEvent]:
        """Get recent emergence events, optionally filtered by type."""
        self._init_emergence_tracking()

        events = self._emergence_events
        if emergence_type:
            events = [e for e in events if e.emergence_type == emergence_type]

        return events[-limit:]

    def get_emergence_stats(self) -> dict:
        """Get summary statistics about emergence detection."""
        self._init_emergence_tracking()

        if not self._emergence_events:
            return {"total_events": 0}

        by_type = {}
        for event in self._emergence_events:
            by_type[event.emergence_type] = by_type.get(event.emergence_type, 0) + 1

        avg_novelty = sum(e.novelty_score for e in self._emergence_events) / len(self._emergence_events)
        avg_impact = sum(e.impact_score for e in self._emergence_events) / len(self._emergence_events)

        return {
            "total_events": len(self._emergence_events),
            "by_type": by_type,
            "avg_novelty_score": avg_novelty,
            "avg_impact_score": avg_impact,
            "active_triggers": len([t for t in self._emergence_triggers if t.is_active]),
            "agents_tracked": len(self._agent_states),
            "metrics_history_size": len(self._metrics_history),
        }
