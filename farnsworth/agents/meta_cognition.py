"""
Farnsworth Meta-Cognition Agent - Self-Reflection and Improvement

Novel Approaches:
1. Capability Gap Detection - Identify where system struggles
2. Strategy Analysis - Evaluate and improve approaches
3. Performance Monitoring - Track quality metrics
4. Improvement Proposals - Generate concrete enhancement ideas

AGI Upgrades:
5. Self-Healing - Anomaly detection with automatic task rerouting
6. Proactive Diagnostics - Curiosity-driven system health checks
7. Adaptive Thresholds - Self-adjusting performance boundaries
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, Callable, Dict, List
from collections import defaultdict
from enum import Enum

from loguru import logger

from farnsworth.agents.base_agent import BaseAgent, AgentCapability, TaskResult

# Nexus integration for self-healing
try:
    from farnsworth.core.nexus import nexus, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    nexus = None


@dataclass
class PerformanceMetric:
    """A tracked performance metric."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)


@dataclass
class CapabilityGap:
    """An identified capability gap."""
    description: str
    severity: float  # 0-1
    frequency: int
    examples: list[str] = field(default_factory=list)
    suggested_improvement: str = ""
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementProposal:
    """A proposed system improvement."""
    title: str
    description: str
    target_area: str  # "model", "prompts", "memory", "agents", "rag"
    priority: float
    expected_impact: str
    implementation_notes: str = ""
    status: str = "proposed"  # proposed, accepted, implemented, rejected
    created_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# SELF-HEALING DATASTRUCTURES (AGI Upgrade)
# =============================================================================

class AnomalyType(Enum):
    """Types of system anomalies."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_SPIKE = "error_spike"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LATENCY_INCREASE = "latency_increase"
    CONFIDENCE_DROP = "confidence_drop"
    CAPABILITY_FAILURE = "capability_failure"
    CASCADE_RISK = "cascade_risk"


class HealingAction(Enum):
    """Self-healing actions."""
    REROUTE_TASK = "reroute_task"
    REDUCE_LOAD = "reduce_load"
    ESCALATE_MODEL = "escalate_model"
    TRIGGER_REFLECTION = "trigger_reflection"
    SPAWN_SPECIALIST = "spawn_specialist"
    CLEAR_CACHE = "clear_cache"
    ADJUST_THRESHOLD = "adjust_threshold"
    NOTIFY_OPERATOR = "notify_operator"


@dataclass
class Anomaly:
    """A detected system anomaly."""
    anomaly_type: AnomalyType
    severity: float  # 0-1
    source: str  # agent/component that triggered
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_action: Optional[HealingAction] = None


@dataclass
class HealingResult:
    """Result of a self-healing action."""
    action: HealingAction
    success: bool
    anomaly_id: str
    details: str
    duration_ms: float = 0.0
    side_effects: List[str] = field(default_factory=list)


@dataclass
class AdaptiveThreshold:
    """Self-adjusting performance threshold."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    adjustment_rate: float = 0.05  # How much to adjust per update
    history: List[float] = field(default_factory=list)
    last_adjusted: datetime = field(default_factory=datetime.now)


class MetaCognitionAgent(BaseAgent):
    """
    Agent for self-reflection and improvement proposals.

    Features:
    - Monitor system performance
    - Detect capability gaps
    - Propose improvements
    - Track what works and what doesn't
    """

    def __init__(
        self,
        reflection_interval_turns: int = 5,
    ):
        super().__init__(
            name="MetaCognition",
            capabilities=[
                AgentCapability.META_COGNITION,
                AgentCapability.REASONING,
                AgentCapability.PLANNING,
            ],
            confidence_threshold=0.5,
        )

        self.reflection_interval = reflection_interval_turns
        self.turns_since_reflection = 0

        # Performance tracking
        self.metrics: list[PerformanceMetric] = []
        self.capability_gaps: list[CapabilityGap] = []
        self.improvement_proposals: list[ImprovementProposal] = []

        # Task outcome history
        self.task_history: list[dict] = []
        self.error_patterns: dict[str, int] = defaultdict(int)

        # Strategy effectiveness
        self.strategy_scores: dict[str, list[float]] = defaultdict(list)

        # Initialize self-healing system (AGI upgrade)
        self._init_self_healing()

    @property
    def system_prompt(self) -> str:
        return """You are the meta-cognitive agent responsible for self-reflection and improvement.

Your role:
1. Analyze system performance and identify issues
2. Detect patterns in failures and limitations
3. Propose concrete improvements
4. Monitor effectiveness of changes

When reflecting:
- Look for recurring failure patterns
- Identify capability gaps
- Evaluate strategy effectiveness
- Suggest actionable improvements

Be specific and constructive in your analysis.
Focus on what can actually be improved."""

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a meta-cognitive task."""
        task_type = self._classify_task(task)

        if task_type == "reflect":
            result = await self._reflect(context)
        elif task_type == "analyze_error":
            result = await self._analyze_error(task, context)
        elif task_type == "propose_improvement":
            result = await self._propose_improvement(task, context)
        elif task_type == "evaluate_strategy":
            result = await self._evaluate_strategy(task, context)
        else:
            result = await self._general_meta_task(task, context)

        return result

    def _classify_task(self, task: str) -> str:
        """Classify the meta-cognitive task type."""
        task_lower = task.lower()

        if "reflect" in task_lower or "review" in task_lower:
            return "reflect"
        elif "error" in task_lower or "failure" in task_lower:
            return "analyze_error"
        elif "improve" in task_lower or "enhance" in task_lower:
            return "propose_improvement"
        elif "strategy" in task_lower or "approach" in task_lower:
            return "evaluate_strategy"

        return "general"

    async def _reflect(self, context: Optional[dict]) -> TaskResult:
        """Perform a reflection on recent performance."""
        # Gather recent data
        recent_tasks = self.task_history[-20:]
        recent_errors = list(self.error_patterns.items())[-10:]
        recent_gaps = self.capability_gaps[-5:]

        # Calculate summary metrics
        if recent_tasks:
            success_rate = sum(1 for t in recent_tasks if t.get("success")) / len(recent_tasks)
            avg_confidence = sum(t.get("confidence", 0) for t in recent_tasks) / len(recent_tasks)
        else:
            success_rate = 0.0
            avg_confidence = 0.0

        # Generate reflection
        reflection_prompt = f"""Reflect on the following system performance:

Success rate: {success_rate:.0%}
Average confidence: {avg_confidence:.0%}
Recent errors: {len(recent_errors)}
Known capability gaps: {len(recent_gaps)}

Recent task outcomes:
{self._format_recent_tasks(recent_tasks)}

Common errors:
{self._format_errors(recent_errors)}

Provide:
1. Overall assessment
2. Key issues identified
3. Recommended actions"""

        response, confidence = await self.generate_response(reflection_prompt, context)

        # Record this reflection
        self.metrics.append(PerformanceMetric(
            name="reflection",
            value=success_rate,
            context={"avg_confidence": avg_confidence},
        ))

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "gaps_detected": len(recent_gaps),
            },
        )

    def _format_recent_tasks(self, tasks: list[dict]) -> str:
        """Format recent tasks for prompt."""
        if not tasks:
            return "No recent tasks"

        lines = []
        for task in tasks[-5:]:
            status = "Success" if task.get("success") else "Failed"
            lines.append(f"- [{status}] {task.get('description', 'Unknown')[:50]}")
        return "\n".join(lines)

    def _format_errors(self, errors: list[tuple]) -> str:
        """Format errors for prompt."""
        if not errors:
            return "No recent errors"

        return "\n".join(f"- {err}: {count} times" for err, count in errors)

    async def _analyze_error(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Analyze an error to understand the root cause."""
        error_info = context.get("error", task) if context else task

        analysis_prompt = f"""Analyze this error/failure:

{error_info}

Provide:
1. Root cause analysis
2. Pattern identification (is this recurring?)
3. Capability gap (what's missing?)
4. Prevention strategy"""

        response, confidence = await self.generate_response(analysis_prompt, context)

        # Record error pattern
        error_type = self._categorize_error(error_info)
        self.error_patterns[error_type] += 1

        # Check if this reveals a capability gap
        if self.error_patterns[error_type] >= 3:
            gap = CapabilityGap(
                description=f"Recurring {error_type} errors",
                severity=min(1.0, self.error_patterns[error_type] / 10),
                frequency=self.error_patterns[error_type],
                examples=[error_info[:100]],
            )
            self.capability_gaps.append(gap)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "error_type": error_type,
                "occurrence_count": self.error_patterns[error_type],
            },
        )

    def _categorize_error(self, error: str) -> str:
        """Categorize an error into a type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower or "context" in error_lower:
            return "context_overflow"
        elif "confidence" in error_lower:
            return "low_confidence"
        elif "hallucin" in error_lower or "incorrect" in error_lower:
            return "accuracy"
        elif "not found" in error_lower or "missing" in error_lower:
            return "missing_information"
        elif "capability" in error_lower or "can't" in error_lower:
            return "capability_limitation"

        return "unknown"

    async def _propose_improvement(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Generate an improvement proposal."""
        # Gather evidence for the proposal
        recent_gaps = self.capability_gaps[-5:]
        recent_errors = dict(list(self.error_patterns.items())[-5:])

        proposal_prompt = f"""Based on the improvement need: {task}

Current issues:
- Capability gaps: {[g.description for g in recent_gaps]}
- Common errors: {recent_errors}

Generate a specific improvement proposal with:
1. Clear title
2. Detailed description
3. Target area (model/prompts/memory/agents/rag)
4. Expected impact
5. Implementation approach"""

        response, confidence = await self.generate_response(proposal_prompt, context)

        # Create structured proposal
        proposal = ImprovementProposal(
            title=task[:50],
            description=response,
            target_area=self._infer_target_area(task, response),
            priority=confidence,
            expected_impact="Based on identified gaps",
        )
        self.improvement_proposals.append(proposal)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "proposal_id": len(self.improvement_proposals) - 1,
                "target_area": proposal.target_area,
            },
        )

    def _infer_target_area(self, task: str, response: str) -> str:
        """Infer the target area for improvement."""
        text = f"{task} {response}".lower()

        if any(kw in text for kw in ["model", "llm", "inference"]):
            return "model"
        elif any(kw in text for kw in ["prompt", "system message", "instruction"]):
            return "prompts"
        elif any(kw in text for kw in ["memory", "context", "recall"]):
            return "memory"
        elif any(kw in text for kw in ["agent", "specialist", "handoff"]):
            return "agents"
        elif any(kw in text for kw in ["retrieval", "search", "rag"]):
            return "rag"

        return "general"

    async def _evaluate_strategy(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Evaluate the effectiveness of a strategy."""
        strategy_name = context.get("strategy", "unknown") if context else "unknown"
        outcomes = context.get("outcomes", []) if context else []

        # Calculate strategy score
        if outcomes:
            success_rate = sum(1 for o in outcomes if o.get("success")) / len(outcomes)
            avg_quality = sum(o.get("quality", 0.5) for o in outcomes) / len(outcomes)
            score = (success_rate + avg_quality) / 2
        else:
            score = 0.5

        self.strategy_scores[strategy_name].append(score)

        eval_prompt = f"""Evaluate strategy: {strategy_name}

Recent performance: {score:.0%}
Historical scores: {self.strategy_scores[strategy_name][-5:]}
Outcomes analyzed: {len(outcomes)}

Provide:
1. Strategy effectiveness assessment
2. Strengths and weaknesses
3. Recommendations for improvement or alternatives"""

        response, confidence = await self.generate_response(eval_prompt, context)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "strategy": strategy_name,
                "current_score": score,
                "historical_avg": sum(self.strategy_scores[strategy_name]) / max(1, len(self.strategy_scores[strategy_name])),
            },
        )

    async def _general_meta_task(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Handle general meta-cognitive tasks."""
        response, confidence = await self.generate_response(task, context)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
        )

    def record_task_outcome(
        self,
        description: str,
        success: bool,
        confidence: float,
        metadata: Optional[dict] = None,
    ):
        """Record a task outcome for analysis."""
        self.task_history.append({
            "description": description,
            "success": success,
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        })

        # Trigger reflection if needed
        self.turns_since_reflection += 1
        if self.turns_since_reflection >= self.reflection_interval:
            self.turns_since_reflection = 0
            # Could trigger async reflection here

        # Keep bounded history
        if len(self.task_history) > 500:
            self.task_history = self.task_history[-250:]

    async def should_reflect(self) -> bool:
        """Check if it's time for reflection."""
        return self.turns_since_reflection >= self.reflection_interval

    def get_capability_gaps(self) -> list[dict]:
        """Get current capability gaps."""
        return [
            {
                "description": gap.description,
                "severity": gap.severity,
                "frequency": gap.frequency,
            }
            for gap in self.capability_gaps
        ]

    def get_improvement_proposals(self, status: Optional[str] = None) -> list[dict]:
        """Get improvement proposals."""
        proposals = self.improvement_proposals
        if status:
            proposals = [p for p in proposals if p.status == status]

        return [
            {
                "title": p.title,
                "target_area": p.target_area,
                "priority": p.priority,
                "status": p.status,
            }
            for p in proposals
        ]

    def get_performance_summary(self) -> dict:
        """Get performance summary."""
        recent = self.task_history[-50:]

        if not recent:
            return {
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "tasks_analyzed": 0,
            }

        return {
            "success_rate": sum(1 for t in recent if t.get("success")) / len(recent),
            "avg_confidence": sum(t.get("confidence", 0) for t in recent) / len(recent),
            "tasks_analyzed": len(recent),
            "capability_gaps": len(self.capability_gaps),
            "pending_proposals": len([p for p in self.improvement_proposals if p.status == "proposed"]),
            "top_error_types": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            "active_anomalies": len([a for a in self.anomalies if not a.resolved]),
            "healing_actions_taken": self.healing_stats.get("total_actions", 0),
        }

    # =========================================================================
    # SELF-HEALING SYSTEM (AGI Upgrade)
    # =========================================================================

    def _init_self_healing(self):
        """Initialize self-healing components."""
        self.anomalies: List[Anomaly] = []
        self.healing_history: List[HealingResult] = []
        self.healing_stats = {
            "total_actions": 0,
            "successful_healings": 0,
            "failed_healings": 0,
        }

        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, AdaptiveThreshold] = {
            "error_rate": AdaptiveThreshold(
                name="error_rate",
                current_value=0.3,  # Alert if >30% errors
                min_value=0.1,
                max_value=0.5,
            ),
            "latency_ms": AdaptiveThreshold(
                name="latency_ms",
                current_value=5000,  # Alert if >5s
                min_value=1000,
                max_value=30000,
            ),
            "confidence_floor": AdaptiveThreshold(
                name="confidence_floor",
                current_value=0.4,  # Alert if confidence drops below
                min_value=0.2,
                max_value=0.7,
            ),
        }

        # Healing action handlers
        self._healing_handlers: Dict[HealingAction, Callable] = {
            HealingAction.REROUTE_TASK: self._heal_reroute_task,
            HealingAction.REDUCE_LOAD: self._heal_reduce_load,
            HealingAction.ESCALATE_MODEL: self._heal_escalate_model,
            HealingAction.TRIGGER_REFLECTION: self._heal_trigger_reflection,
            HealingAction.ADJUST_THRESHOLD: self._heal_adjust_threshold,
        }

        # Rerouting callbacks (set by swarm orchestrator)
        self._reroute_callback: Optional[Callable] = None
        self._escalate_callback: Optional[Callable] = None

    async def detect_anomalies(self) -> List[Anomaly]:
        """
        Proactive anomaly detection across system metrics.

        Checks:
        - Error rate spikes
        - Performance degradation
        - Confidence drops
        - Capability failures
        """
        detected = []
        recent_tasks = self.task_history[-20:]

        if len(recent_tasks) < 5:
            return detected  # Not enough data

        # 1. Error rate check
        error_rate = sum(1 for t in recent_tasks if not t.get("success")) / len(recent_tasks)
        threshold = self.adaptive_thresholds["error_rate"].current_value

        if error_rate > threshold:
            severity = min(1.0, error_rate / threshold - 1.0 + 0.5)
            anomaly = Anomaly(
                anomaly_type=AnomalyType.ERROR_SPIKE,
                severity=severity,
                source="task_history",
                description=f"Error rate {error_rate:.0%} exceeds threshold {threshold:.0%}",
                evidence={"error_rate": error_rate, "threshold": threshold, "sample_size": len(recent_tasks)},
            )
            detected.append(anomaly)
            self.anomalies.append(anomaly)

        # 2. Confidence degradation check
        avg_confidence = sum(t.get("confidence", 0.5) for t in recent_tasks) / len(recent_tasks)
        conf_floor = self.adaptive_thresholds["confidence_floor"].current_value

        if avg_confidence < conf_floor:
            severity = min(1.0, (conf_floor - avg_confidence) / conf_floor + 0.3)
            anomaly = Anomaly(
                anomaly_type=AnomalyType.CONFIDENCE_DROP,
                severity=severity,
                source="task_history",
                description=f"Avg confidence {avg_confidence:.0%} below floor {conf_floor:.0%}",
                evidence={"avg_confidence": avg_confidence, "floor": conf_floor},
            )
            detected.append(anomaly)
            self.anomalies.append(anomaly)

        # 3. Capability failure pattern check
        capability_errors = [t for t in recent_tasks if t.get("metadata", {}).get("error_type") == "capability_limitation"]
        if len(capability_errors) >= 3:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.CAPABILITY_FAILURE,
                severity=0.7,
                source="capability_tracking",
                description=f"Repeated capability failures ({len(capability_errors)} in last {len(recent_tasks)} tasks)",
                evidence={"failure_count": len(capability_errors)},
            )
            detected.append(anomaly)
            self.anomalies.append(anomaly)

        # 4. Cascade risk detection (multiple simultaneous issues)
        if len(detected) >= 2:
            cascade_anomaly = Anomaly(
                anomaly_type=AnomalyType.CASCADE_RISK,
                severity=min(1.0, sum(a.severity for a in detected) / 2),
                source="anomaly_aggregation",
                description=f"Multiple concurrent anomalies detected ({len(detected)}), cascade risk",
                evidence={"anomaly_types": [a.anomaly_type.value for a in detected]},
            )
            detected.append(cascade_anomaly)
            self.anomalies.append(cascade_anomaly)

        # Emit anomaly signals via Nexus
        if NEXUS_AVAILABLE and nexus and detected:
            for anomaly in detected:
                asyncio.create_task(nexus.emit(
                    SignalType.ANOMALY_DETECTED,
                    {
                        "anomaly_type": anomaly.anomaly_type.value,
                        "severity": anomaly.severity,
                        "source": anomaly.source,
                        "description": anomaly.description,
                    },
                    source="meta_cognition",
                    urgency=0.5 + anomaly.severity * 0.5,
                ))

        return detected

    async def self_heal(self, anomaly: Anomaly) -> HealingResult:
        """
        Attempt to self-heal from an anomaly.

        Selects appropriate healing action based on anomaly type
        and executes the corresponding handler.
        """
        import time
        start_time = time.time()

        # Select healing action based on anomaly type
        action = self._select_healing_action(anomaly)

        # Execute healing
        handler = self._healing_handlers.get(action)
        success = False
        details = ""

        if handler:
            try:
                success, details = await handler(anomaly)
            except Exception as e:
                details = f"Healing failed with error: {e}"
                logger.error(f"Self-healing error: {e}")
        else:
            details = f"No handler for action {action.value}"

        # Record result
        duration_ms = (time.time() - start_time) * 1000
        result = HealingResult(
            action=action,
            success=success,
            anomaly_id=str(id(anomaly)),
            details=details,
            duration_ms=duration_ms,
        )

        self.healing_history.append(result)
        self.healing_stats["total_actions"] += 1
        if success:
            self.healing_stats["successful_healings"] += 1
            anomaly.resolved = True
            anomaly.resolution_action = action
        else:
            self.healing_stats["failed_healings"] += 1

        logger.info(f"Self-healing {'succeeded' if success else 'failed'}: {action.value} - {details}")

        return result

    def _select_healing_action(self, anomaly: Anomaly) -> HealingAction:
        """Select appropriate healing action for an anomaly."""
        action_map = {
            AnomalyType.ERROR_SPIKE: HealingAction.REROUTE_TASK,
            AnomalyType.PERFORMANCE_DEGRADATION: HealingAction.REDUCE_LOAD,
            AnomalyType.CONFIDENCE_DROP: HealingAction.ESCALATE_MODEL,
            AnomalyType.CAPABILITY_FAILURE: HealingAction.SPAWN_SPECIALIST,
            AnomalyType.CASCADE_RISK: HealingAction.REDUCE_LOAD,
            AnomalyType.LATENCY_INCREASE: HealingAction.CLEAR_CACHE,
        }

        action = action_map.get(anomaly.anomaly_type, HealingAction.TRIGGER_REFLECTION)

        # High severity always triggers reflection
        if anomaly.severity >= 0.8:
            action = HealingAction.TRIGGER_REFLECTION

        return action

    async def _heal_reroute_task(self, anomaly: Anomaly) -> tuple[bool, str]:
        """Reroute tasks away from failing component."""
        if self._reroute_callback:
            try:
                await self._reroute_callback(anomaly.source)
                return True, f"Tasks rerouted away from {anomaly.source}"
            except Exception as e:
                return False, f"Reroute failed: {e}"
        return False, "No reroute callback configured"

    async def _heal_reduce_load(self, anomaly: Anomaly) -> tuple[bool, str]:
        """Reduce system load to recover from degradation."""
        # Emit signal to reduce load across the system
        if NEXUS_AVAILABLE and nexus:
            await nexus.emit(
                SignalType.EXTERNAL_EVENT,
                {
                    "event": "load_reduction_requested",
                    "reason": anomaly.description,
                    "severity": anomaly.severity,
                },
                source="meta_cognition_healing",
                urgency=0.7,
            )
            return True, "Load reduction signal emitted"
        return False, "Nexus not available for load reduction"

    async def _heal_escalate_model(self, anomaly: Anomaly) -> tuple[bool, str]:
        """Escalate to a more capable model."""
        if self._escalate_callback:
            try:
                await self._escalate_callback("confidence_issue")
                return True, "Escalated to higher-capability model"
            except Exception as e:
                return False, f"Escalation failed: {e}"
        return False, "No escalation callback configured"

    async def _heal_trigger_reflection(self, anomaly: Anomaly) -> tuple[bool, str]:
        """Trigger deep reflection on the issue."""
        result = await self._reflect({"anomaly": anomaly.description})
        return True, f"Reflection triggered: {result.output[:100]}..."

    async def _heal_adjust_threshold(self, anomaly: Anomaly) -> tuple[bool, str]:
        """Adjust adaptive threshold based on evidence."""
        # Find which threshold needs adjustment
        threshold_name = anomaly.evidence.get("threshold_name")
        if threshold_name and threshold_name in self.adaptive_thresholds:
            threshold = self.adaptive_thresholds[threshold_name]
            self._adjust_threshold(threshold, direction="relax")
            return True, f"Adjusted {threshold_name} threshold"
        return False, "No threshold to adjust"

    def _adjust_threshold(self, threshold: AdaptiveThreshold, direction: str = "auto"):
        """
        Adjust an adaptive threshold based on recent performance.

        direction: "tighten", "relax", or "auto"
        """
        recent_values = threshold.history[-10:] if threshold.history else []

        if direction == "auto" and recent_values:
            avg = sum(recent_values) / len(recent_values)
            if avg > threshold.current_value * 1.2:
                direction = "relax"  # Values consistently high, relax threshold
            elif avg < threshold.current_value * 0.8:
                direction = "tighten"  # Values consistently low, can tighten

        adjustment = threshold.current_value * threshold.adjustment_rate

        if direction == "relax":
            threshold.current_value = min(threshold.max_value, threshold.current_value + adjustment)
        elif direction == "tighten":
            threshold.current_value = max(threshold.min_value, threshold.current_value - adjustment)

        threshold.last_adjusted = datetime.now()
        logger.debug(f"Adjusted threshold {threshold.name} to {threshold.current_value:.3f}")

    def set_reroute_callback(self, callback: Callable):
        """Set callback for task rerouting."""
        self._reroute_callback = callback

    def set_escalate_callback(self, callback: Callable):
        """Set callback for model escalation."""
        self._escalate_callback = callback

    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run proactive health check (curiosity-driven diagnostics).

        Returns comprehensive health report.
        """
        health = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {},
        }

        # 1. Anomaly detection
        anomalies = await self.detect_anomalies()
        health["checks"]["anomaly_detection"] = {
            "passed": len(anomalies) == 0,
            "anomalies_found": len(anomalies),
            "details": [a.description for a in anomalies],
        }

        # 2. Performance check
        perf = self.get_performance_summary()
        health["checks"]["performance"] = {
            "passed": perf.get("success_rate", 0) >= 0.7,
            "success_rate": perf.get("success_rate", 0),
            "avg_confidence": perf.get("avg_confidence", 0),
        }

        # 3. Capability health
        health["checks"]["capabilities"] = {
            "passed": len(self.capability_gaps) < 5,
            "gap_count": len(self.capability_gaps),
            "top_gaps": [g.description for g in self.capability_gaps[:3]],
        }

        # 4. Self-healing effectiveness
        healing_rate = (
            self.healing_stats["successful_healings"] / max(1, self.healing_stats["total_actions"])
        )
        health["checks"]["self_healing"] = {
            "passed": healing_rate >= 0.5 or self.healing_stats["total_actions"] == 0,
            "success_rate": healing_rate,
            "total_actions": self.healing_stats["total_actions"],
        }

        # Overall status
        failed_checks = sum(1 for c in health["checks"].values() if not c.get("passed", True))
        if failed_checks >= 3:
            health["status"] = "critical"
        elif failed_checks >= 1:
            health["status"] = "degraded"

        # Auto-heal critical issues
        if health["status"] == "critical" and anomalies:
            for anomaly in anomalies[:2]:  # Heal top 2 anomalies
                await self.self_heal(anomaly)

        return health

    def get_healing_stats(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        return {
            **self.healing_stats,
            "active_anomalies": len([a for a in self.anomalies if not a.resolved]),
            "resolved_anomalies": len([a for a in self.anomalies if a.resolved]),
            "adaptive_thresholds": {
                name: {
                    "current": t.current_value,
                    "min": t.min_value,
                    "max": t.max_value,
                }
                for name, t in self.adaptive_thresholds.items()
            },
            "recent_healings": [
                {
                    "action": h.action.value,
                    "success": h.success,
                    "duration_ms": h.duration_ms,
                }
                for h in self.healing_history[-5:]
            ],
        }
